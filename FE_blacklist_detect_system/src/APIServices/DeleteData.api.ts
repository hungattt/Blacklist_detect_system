import customAxios from "../utils/customAxios";

type RequestOptions = {
  headers: {
    "Content-Type": string;
  };
};

export const deleteData = async ({
  data,
  options,
}: {
  data: any;
  options: RequestOptions;
}) => {
  try {
    // Log FormData content manually
    for (const [key, value] of data.entries()) {
      console.log(key, value);
    }
    const response = await customAxios.delete("delete_data/", {
      data, // include your data here
      ...options,
    });
    return response;
  } catch (error) {
    console.error("deleteDataError: ", error);
  }
};

export const postData = async () => {
  try {
    const response = await customAxios.post("delete_data/");
    return response.data;
  } catch (error) {
    console.error("postDeleteDataError: ", error);
  }
};
