import { Link } from "react-router-dom";
import "./css/ErrorPage.css";

const ErrorPage = () => {
  return (
    <div id="notfound">
      <div className="notfound">
        <div className="notfound-404">
          <h1>:(</h1>
        </div>

        <div>
          <h2>404 - Trang không tồn tại</h2>
          <p>
            Trang bạn đang tìm kiếm có thể đã bị xóa, đổi tên hoặc tạm thời
            không có sẵn.
          </p>
          <Link to={"/Flight"}>Trang Chủ</Link>
        </div>
      </div>
    </div>
  );
};

export default ErrorPage;
