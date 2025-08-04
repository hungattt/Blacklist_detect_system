import customAxios from "../utils/customAxios";

type RequestOptions = {
  headers: {
    "Content-Type": string;
  };
};

type updateDataCreateDataService = {
  avata: File | null | undefined;
  name: string;
  full_name: string;
  year_of_birth: number;
  hometown: string;
  id_number: string;
  violation: string;
  data: File | File[] | null | undefined;
};

export const updateData = async ({
  data,
  options,
}: {
  data: updateDataCreateDataService;
  options: RequestOptions;
}) => {
  try {
    const response = await customAxios.patch("update_data/", data, options);
    return response;
  } catch (error) {
    console.error("updateDataCreateDataServiceError: ", error);
  }
};
